import time
import onnxruntime
import numpy as np
import open3d as o3d
import torch
from ops import VoxelizationByGridShape
from utils.anchor_3d_generator import Anchor3DRangeGenerator
from utils.show_lidar_box import show_box

onnx_file_path = "end2end.onnx"


def py_cpu_nms(dets, scores, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 打分从大到小排列，取index
    order = np.asarray(reversed(scores.argsort()[::1]))
    # keep为最后保留的边框
    keep = []
    while order.size > 0:
        # order(0)是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)
        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= thresh)[0]
        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]
    return keep


class PointPillars(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = init_model(config_file, checkpoint_file, device='cpu')
        self.onnx_session = onnxruntime.InferenceSession(onnx_file_path, providers=['CPUExecutionProvider'])
        self.box_code_size = 7
        self.num_classes = 1
        self.nms_pre = 10
        self.max_num = 4
        self.score_thr = 0.1
        self.nms_thr = 0.01
        self.voxel_layer = VoxelizationByGridShape(voxel_size=[0.16, 0.16, 10.64],
                                                   point_cloud_range=[0, -39.68, -3.52, 69.12, 39.68, 7.12],
                                                   max_num_points=64)
        self.anchor3d_generater = Anchor3DRangeGenerator(
            [[0, -39.68, -0.18, 69.12, 39.68, -0.18]],
            sizes=[[25.5, 1.78, 4.02]],
            scales=[1],
            rotations=[0, 1.57],
            reshape_out=True,
            size_per_range=True
        )
        self.mlvl_priors = self.anchor3d_generater.grid_anchors([(248, 216)], device="cpu")
        self.mlvl_priors = [prior.reshape(-1, self.box_code_size) for prior in self.mlvl_priors]

    def pre_process(self, x):
        res_voxels, res_coors, res_num_points = self.voxel_layer(x)
        return res_voxels, res_coors, res_num_points

    def xywhr2xyxyr(self, boxes_xywhr):
        boxes = torch.zeros_like(boxes_xywhr)
        half_w = boxes_xywhr[..., 2] / 2
        half_h = boxes_xywhr[..., 3] / 2
        boxes[..., 0] = boxes_xywhr[..., 0] - half_w
        boxes[..., 1] = boxes_xywhr[..., 1] - half_h
        boxes[..., 2] = boxes_xywhr[..., 0] + half_w
        boxes[..., 3] = boxes_xywhr[..., 1] + half_h
        boxes[..., 4] = boxes_xywhr[..., 4]
        return boxes

    def box3d_multiclass_nms(self, mlvl_bboxes, mlvl_bboxes_for_nms, mlvl_scores, mlvl_dir_scores):
        num_classes = mlvl_scores.shape[1] - 1
        bboxes = []
        scores = []
        labels = []
        dir_scores = []
        for i in range(0, num_classes):
            cls_inds = mlvl_scores[:, i] > self.score_thr
            if not cls_inds.any():
                continue
            _scores = mlvl_scores[cls_inds, i]
            _bboxes_for_nms = mlvl_bboxes_for_nms[cls_inds, :]
            selected = py_cpu_nms(_bboxes_for_nms, _scores, self.nms_thr)
            bboxes.append(mlvl_bboxes[selected])
            scores.append(_scores[selected])
            cls_label = mlvl_bboxes.new_full((len(selected),), i, dtype=torch.long)
            labels.append(cls_label)
            dir_scores.append(mlvl_dir_scores[selected])
        if bboxes:
            bboxes = torch.cat(bboxes, dim=0)
            scores = torch.cat(scores, dim=0)
            labels = torch.cat(labels, dim=0)
            dir_scores = torch.cat(dir_scores, dim=0)
            if bboxes.shape[0] > self.max_num:
                _, inds = scores.sort(descending=True)
                inds = inds[:self.max_num]
                bboxes = bboxes[inds, :]
                labels = labels[inds]
                scores = scores[inds]
                dir_scores = dir_scores[inds]
        else:
            bboxes = mlvl_scores.new_zeros((0, mlvl_bboxes.size(-1)))
            scores = mlvl_scores.new_zeros((0,))
            labels = mlvl_scores.new_zeros((0,), dtype=torch.long)
            dir_scores = mlvl_scores.new_zeros((0,))
        return bboxes, scores, labels, dir_scores

    def decode(self, anchors, deltas):
        xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt = torch.split(deltas, 1, dim=-1)
        za = za + ha / 2
        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za
        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
        rg = rt + ra
        zg = zg - hg / 2
        return torch.cat([xg, yg, zg, wg, lg, hg, rg], dim=-1)

    def obb2aabb(self, pcd, obboxes):
        aabb_bev_boxes = []
        for data in obboxes:
            center = data[:3]
            extent = data[3: 6]
            # 定义旋转轴和旋转角度（使用弧度）
            axis = np.array([0, 0, data[6]])  # 例如，绕x轴旋转
            # 从轴角获取旋转矩阵
            R = o3d.geometry.get_rotation_matrix_from_xyz(axis)
            obbox = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)
            inside_points = pcd.select_by_index(obbox.get_point_indices_within_bounding_box(pcd.points))
            box = inside_points.get_axis_aligned_bounding_box()
            aabb_center = box.get_center().tolist()
            aabb_scale = box.get_extent().tolist()
            bev_box = aabb_center[:2] + aabb_scale[:2] + [data[6]]
            aabb_bev_boxes.append(bev_box)
        return torch.tensor(aabb_bev_boxes)

    def predict_by_feat_single(self, pcd, cls_score, bbox_pred, dir_cls_pred):
        priors = self.mlvl_priors[0]
        dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        dir_cls_scores = torch.max(dir_cls_pred, dim=-1)[1]
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes)
        scores = cls_score.sigmoid()
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, self.box_code_size)
        max_scores, _ = scores.max(dim=1)
        _, topk_inds = max_scores.topk(self.nms_pre)
        priors = priors[topk_inds, :].cpu()
        bbox_pred = bbox_pred[topk_inds, :]
        scores = scores[topk_inds, :]
        dir_cls_scores = dir_cls_scores[topk_inds]
        bboxes = self.decode(priors, bbox_pred)
        aabboxes = self.obb2aabb(pcd, bboxes)
        # mlvl_bboxes_bev = torch.cat([bboxes[:, 0:2], bboxes[:, 3:5], bboxes[:, 6:]], dim=1)
        mlvl_bboxes_for_nms = self.xywhr2xyxyr(aabboxes)
        padding = scores.new_zeros(scores.shape[0], 1)
        scores = torch.cat([scores, padding], dim=1)
        results = self.box3d_multiclass_nms(bboxes, mlvl_bboxes_for_nms, scores, dir_cls_scores)
        bboxes, scores, labels, dir_scores = results
        if bboxes.shape[0] > 0:
            dir_rot = bboxes[..., 6] + np.pi / 2 - torch.floor(bboxes[..., 6] + np.pi / 2 / np.pi) * np.pi
            bboxes[..., 6] = (dir_rot - np.pi / 2 + np.pi * dir_scores.to(bboxes.dtype))
        return bboxes, scores, labels


pointpillars = PointPillars()


def pointpillars_forward_and_show_result(pcd_file_path='000000.bin'):
    if pcd_file_path.endswith(".bin"):
        points = np.fromfile(pcd_file_path, dtype=np.float32)
    elif pcd_file_path.endswith(".pcd"):
            bg_cloud = o3d.io.read_point_cloud(pcd_file_path)
            new_array = np.asarray(bg_cloud.points, dtype=np.float32)
            xyz = new_array[:, :3]
            points = np.insert(xyz, 3, np.ones(xyz.shape[0]), axis=1)
    points = torch.from_numpy(points.reshape(-1, 4))
    xyz = points[:, :3]
    # 创建 PointCloud 对象
    pcd = o3d.geometry.PointCloud()
    # 将点坐标数据赋值给 PointCloud 对象
    pcd.points = o3d.utility.Vector3dVector(xyz)
    start_time = time.time()
    res_voxels, res_coors, res_num_points = pointpillars.voxel_layer(points)
    res_coors = torch.cat((torch.zeros([res_coors.shape[0], 1]), res_coors), dim=1)
    inputs = {
            'voxels': res_voxels.numpy(),
            'num_points': res_num_points.type(torch.int32).numpy(),
            'coors': res_coors.type(torch.int32).numpy()
            }
    results = pointpillars.onnx_session.run(None, inputs)
    result = pointpillars.predict_by_feat_single(pcd,
                                                 torch.from_numpy(results[0]).squeeze(),
                                                 torch.from_numpy(results[1]).squeeze(),
                                                 torch.from_numpy(results[2]).squeeze())
    print("pointpillars use time:{}ms".format((time.time() - start_time) * 1000))
    show_box(points, result[0])
    print(result)


if __name__ == '__main__':
    file_path = "*****.pcd"
    pointpillars_forward_and_show_result()

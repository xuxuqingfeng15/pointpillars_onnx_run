import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes
from sklearn.decomposition import PCA


def compute_length_width(pixel_coordinates):
    """
    此函数通过主成分分析计算像素点集合的长和宽
    :param pixel_coordinates: 形状为 (n, 2) 的 numpy 数组，包含像素点的 (x, y) 坐标
    :return: 长和宽
    """
    # 计算几何中心
    geometric_center = np.mean(pixel_coordinates, axis=0)
    # 实例化 PCA 对象，设置主成分数量为 2
    pca = PCA(n_components=2)
    # 对像素点坐标进行 PCA 变换
    pca.fit(pixel_coordinates)
    # 获取特征值
    # 对数据进行 PCA 变换
    transformed_coords = pca.transform(pixel_coordinates)
    length = transformed_coords[:, 0].max() - transformed_coords[:, 0].min()
    width = transformed_coords[:, 1].max() - transformed_coords[:, 1].min()
    return length, width, geometric_center


def pca_visualization(pixel_coordinates):
    """
    此函数进行 PCA 分析并可视化结果
    :param pixel_coordinates: 像素坐标数组，形状为 (n, 2)
    """
    # 确保输入是 numpy 数组
    pixel_coordinates = np.array(pixel_coordinates)
    # 创建 PCA 对象并拟合数据
    pca = PCA(n_components=2)
    pca.fit(pixel_coordinates)
    # 对数据进行 PCA 变换
    transformed_coords = pca.transform(pixel_coordinates)
    # 获取主成分的特征向量
    eigenvectors = pca.components_
    # 获取主成分的特征值
    eigenvalues = pca.explained_variance_
    # 绘制原始数据
    plt.scatter(pixel_coordinates[:, 0], pixel_coordinates[:, 1], color='blue', label='Original Data')
    # 绘制主成分方向
    mean = np.mean(pixel_coordinates, axis=0)
    for eigenvalue, eigenvector in zip(eigenvalues, eigenvectors):
        std_dev = np.sqrt(eigenvalue)
        # 绘制方向线，长度为 2 倍标准差
        end_point = mean + 2 * std_dev * eigenvector
        plt.arrow(mean[0], mean[1], end_point[0] - mean[0], end_point[1] - mean[1],
                 head_width=0.2, head_length=0.2, fc='red', ec='red', label='Principal Components')
    # 绘制投影后的数据
    plt.scatter(transformed_coords[:, 0], transformed_coords[:, 1], color='green', label='Transformed Data')
    # 设置图例
    plt.legend()
    # 显示图形
    plt.show()


def show_box(points, boxes, problem_png_dir=None):
    points = np.asarray(points)
    visualizer = Det3DLocalVisualizer()
    # set point cloud in visualizer
    visualizer.set_points(points)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:, :3]))
    draw_list = [pcd]
    for data in boxes:
        # obbox框,将obbox转换为axis—box
        extent = data[3: 6]
        center = np.array(data[:3])
        center[2] = center[2] + extent[2]/2
        # 定义旋转轴和旋转角度（使用弧度）
        axis = np.array([0, 0, data[6]])  # 例如，绕x轴旋转
        # 从轴角获取旋转矩阵
        R = o3d.geometry.get_rotation_matrix_from_xyz(axis)
        obbox = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)
        inside_points = pcd.select_by_index(obbox.get_point_indices_within_bounding_box(pcd.points))
        # 获取这些点的x,y坐标
        # obbox_xy = np.asarray(inside_points.points)[:, :2]
        # length, width, geometric_center = compute_length_width(obbox_xy)
        # pca_visualization(obbox_xy)
        box = inside_points.get_axis_aligned_bounding_box()
        # bbox_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box)
        bbox_lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obbox)
        bbox_lines.paint_uniform_color([0, 1.0, 0])
        draw_list.append(bbox_lines)
        new_inside_points = pcd.select_by_index(box.get_point_indices_within_bounding_box(pcd.points))
        new_inside_points.paint_uniform_color([1.0, 0, 0])
        # o3d.visualization.draw([new_inside_points, bbox_lines])
    o3d.visualization.draw(draw_list)


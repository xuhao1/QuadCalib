import yaml
import numpy as np

def load_aprilgrid_config(yaml_file):
    try:
        """加载 AprilGrid 配置文件"""
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
        
        target_type = config.get('target_type', 'aprilgrid')
        if target_type != 'aprilgrid':
            raise ValueError(f"Unsupported target type: {target_type}")
    except Exception as e:
        print(f"Error: {e}, will load default configuration.")
        config = {
            'tagCols': 6,
            'tagRows': 6,
            'tagSize': 0.088,
            'tagSpacing': 0.3
        }    
    return config

def generate_aprilgrid_3d_points(tagCols, tagRows, tagSize, tagSpacing):
    """
    根据配置生成 AprilGrid 的 3D 点，并以 {id: [[pt1], [pt2], [pt3], [pt4]]} 的形式返回
    """
    grid_points = {}
    tag_stride = tagSize * (1 + tagSpacing)  # 标签中心之间的间距
    tag_id = 0  # 初始标签 ID
    for row in range(tagRows):
        for col in range(tagCols):
            # 计算当前标签的 4 个角点坐标
            # convert corners to cv::Mat (4 consecutive corners form one tag)
            # point ordering here in OpenCV
            #           2-------3
            #     y     | TAG 0 |
            #    ^      1-------0
            #    |-->x
            tag_origin_x = col * tag_stride
            tag_origin_y = row * tag_stride
            tag_corners = [
                [tag_origin_x + tagSize, tag_origin_y, 0],  # 0
                [tag_origin_x, tag_origin_y, 0],  # 1
                [tag_origin_x, tag_origin_y + tagSize, 0],  # 2
                [tag_origin_x + tagSize, tag_origin_y + tagSize, 0],  # 3
            ]
            grid_points[tag_id] = np.array(tag_corners, dtype=np.float32)
            tag_id += 1  # 更新标签 ID
    
    return grid_points


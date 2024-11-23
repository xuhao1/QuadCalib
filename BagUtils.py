from bagpy import bagreader

def CountBagImageNumber(bag_path):
    b = bagreader(bag_path)
    df = b.topic_table
    image_types = ["sensor_msgs/Image", "sensor_msgs/CompressedImage"]
    filtered_df = df[df["Types"].isin(image_types)]
    total_image_count = filtered_df["Message Count"].sum()
    return total_image_count
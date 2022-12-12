from dataclasses import dataclass


@dataclass
class TargetTrackingOutput:
    pass

    def write_to_file(self, output_file: str):
        pass


# def tt_parser(file: h5py.File):
#     data = file["target_tracking_output"]["data"]
#     header_time = np.array(file["target_tracking_output"]["zeader"]["timestamp_ns"])
#     for i in range(len(header_time)):
#         timestamp = data["timestamp"]["nanoseconds"]["value"][i]
#         classification = data["object_attributes"]["classification"][i]
#         existance_probs = data["object_attributes"]["existence_probability"][i]
#         ids = data["object_attributes"]["id"][i]
#         lengths = data["object_boxes"]["length"][i]
#         widths = data["object_boxes"]["width"][i]
#         heights = data["object_boxes"]["height"][i]
#         lat_pos = data["object_states"]["lat_pos"][i]
#         lon_pos = data["object_states"]["lon_pos"][i]
#         heading = data["object_states"]["heading"][i]
#         speed = data["object_states"]["speed"][i]

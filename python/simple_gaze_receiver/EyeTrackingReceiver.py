# EyeTrackingReceiver.py
import zmq

class EyeTrackingReceiver:
    def __init__(self, local_ip, remote_ip, port, use_remote, shared_data):
        self.lip = local_ip
        self.rip = remote_ip
        self.p = port
        self.status = use_remote
        self.BB = 0.0
        self.BE = 0.0
        self.eye_detected = True
        self.shared_data = shared_data

        # Initialize ZMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.RCVHWM, 0)

        # Connect to appropriate IP
        if self.status:
            self.socket.connect(f"tcp://{self.rip}:{self.p}")
        else:
            self.socket.connect(f"tcp://{self.lip}:{self.p}")
        print(
            f"Attempting to connect to {'remote' if self.status else 'local'} server at {'tcp://' + self.rip + ':' + str(self.p) if self.status else 'tcp://' + self.lip + ':' + str(self.p)}"
        )

    def receive_data(self):
        while True:
            if "stop" in self.shared_data and self.shared_data["stop"].value:
                print("EyeTrackingReceiver: Stop flag detected, exiting...")
                break

            try:
                while True:  # Process all messages in the buffer
                    text = self.socket.recv_string(flags=zmq.NOBLOCK)
                    if text:
                        self.parse_data(text)
                        #print(text, end="\n")
            except zmq.error.Again:
                pass

            except KeyboardInterrupt:
                break

        self.socket.close()
        self.context.term()
        print("EyeTrackingReceiver stopped.")

    def parse_data(self, data_text):
        data = data_text.split(";")
        # print(f"data: {data}")
        try:
            server_data = {
                "ID": float(data[0]),
                "Timestamp": float(data[1]),
                "PicNum": int(data[11]),
                "GazeX": float(data[2]),
                "GazeY": float(data[3]),
                "PupilSizeLeft": float(data[4]),
                "PupilSizeRight": float(data[5]),
                "RScore": float(data[9]),
                "LScore": float(data[10]),
                "eyeEvent": str(data[20]),
            }

            if data[20] == " NA":
                if self.eye_detected:
                    print("eyes are not detected")
                    self.eye_detected = False
            else:
                if not self.eye_detected:
                    print("eyes are detected, analyze start")
                    self.eye_detected = True
                self.shared_data["ID"].value = server_data["ID"]
                self.shared_data["Timestamp"].value = server_data["Timestamp"]
                self.shared_data["PicNum"].value = server_data["PicNum"]
                self.shared_data["GazeX"].value = server_data["GazeX"]
                self.shared_data["GazeY"].value = server_data["GazeY"]
                self.shared_data["PupilSizeLeft"].value = server_data["PupilSizeLeft"]
                self.shared_data["PupilSizeRight"].value = server_data["PupilSizeRight"]
                self.shared_data["RScore"].value = server_data["RScore"]
                self.shared_data["LScore"].value = server_data["LScore"]
                self.shared_data["eyeEvent"].value = server_data["eyeEvent"]

        except Exception as e:
            print(f"Error parsing data: {e}")
            pass

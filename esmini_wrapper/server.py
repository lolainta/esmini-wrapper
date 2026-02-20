import grpc
from concurrent import futures
import os
import logging

from google.protobuf.json_format import MessageToDict

from sbsvf_api import sim_server_pb2, sim_server_pb2_grpc
from sbsvf_api.pong_pb2 import Pong
from sbsvf_api.empty_pb2 import Empty
from sbsvf_api.scenario_pb2 import ScenarioPack
from sbsvf_api.object_pb2 import (
    ObjectState,
    ObjectKinematic,
    Shape,
    ShapeType,
    RoadObjectType,
)
from sbsvf_api.control_pb2 import CtrlCmd, CtrlMode


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)


class EsminiService(sim_server_pb2_grpc.SimServerServicer):
    def __init__(self):
        pass

    def Ping(self, request, context):
        logger.info(f"Received ping from client: {context.peer()}")
        return Pong(msg="Esmini alive")

    def Init(self, request, context):
        pass

    def Reset(self, request, context):
        pass

    def Step(self, request, context):
        pass

    def Stop(self, request, context):
        pass

    def ShouldQuit(self, request, context):
        pass


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))

    sim_server_pb2_grpc.add_SimServerServicer_to_server(EsminiService(), server)

    PORT = os.environ.get("PORT", "50051")

    server.add_insecure_port(f"[::]:{PORT}")
    server.start()

    print(f"gRPC server running on port {PORT}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("Shutting down gRPC server...")
        server.stop(0)


if __name__ == "__main__":
    serve()

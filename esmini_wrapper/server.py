from pprint import pprint

import grpc
from concurrent import futures
import os
import logging

from google.protobuf.json_format import MessageToDict

from sbsvf_api import sim_server_pb2, sim_server_pb2_grpc
from sbsvf_api.pong_pb2 import Pong
from sbsvf_api.empty_pb2 import Empty
from esmini import EsminiAdapter

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)


class EsminiService(sim_server_pb2_grpc.SimServerServicer):
    def __init__(self):
        self._esmini = None

    def Ping(self, request, context):
        logger.info(f"Received ping from client: {context.peer()}")
        return Pong(msg="Esmini alive")

    def Init(self, request, context):
        cfg = request.config.config
        config = MessageToDict(cfg)
        output_dir = request.output_dir.path
        self.dt = request.dt

        if self._esmini is None:
            self._esmini = EsminiAdapter(output_dir, config)
        pprint(config)

        return sim_server_pb2.SimServerMessages.InitResponse(
            success=True, msg="Esmini initialized"
        )

    def Reset(self, request, context):
        output_dir = request.output_dir.path
        sps = request.scenario_pack
        params = request.params
        objects = self._esmini.reset(output_dir, sps, params)
        return sim_server_pb2.SimServerMessages.ResetResponse(objects=objects)

    def Step(self, request, context):
        ctrl_cmd = request.ctrl_cmd
        timestamp_ns = request.timestamp_ns
        objects = self._esmini.step(ctrl_cmd, timestamp_ns)
        return sim_server_pb2.SimServerMessages.StepResponse(objects=objects)

    def Stop(self, request, context):
        self._esmini.stop()
        return Empty()

    def ShouldQuit(self, request, context):
        return sim_server_pb2.SimServerMessages.ShouldQuitResponse(
            should_quit=self._esmini.should_quit()
        )


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

"""This module implements a live inference client for Triton backends."""

import sleap
import numpy as np
import attr
import tritonclient.grpc as grpcclient
from tritonclient.grpc import service_pb2

from typing import Optional, List, Tuple, Dict


@attr.s(auto_attribs=True)
class InferenceClient():
    """Triton inference client for live predictions.

    Attributes:
        url: URL and port of the Triton endpoint. Defaults to "localhost:8001".
        model_name: Name of the model in the model repository.
        model_version: Version of hte model in the model repository.
    """
    url: str = "localhost:8001"
    model_name: str = "trtmodel_FP16"
    model_version: str = "1"
    _client: Optional[grpcclient.InferenceServerClient] = None
    _model_metadata:  Optional[service_pb2.ModelMetadataResponse] = None
    _model_config: Optional[service_pb2.ModelConfigResponse] = None
    _request_count: int = 0

    def connect(self):
        """Connect to the Triton server and query the model metadata."""
        self._client = grpcclient.InferenceServerClient(url=self.url)
        self._model_metadata = self._client.get_model_metadata(
            model_name=self.model_name,
            model_version=self.model_version
        )
        self._model_config = self._client.get_model_config(
            model_name=self.model_name,
            model_version=self.model_version
        )

    @property
    def input_names(self) -> List[str]:
        """Return the names of the input tensors of the model."""
        return [x.name for x in self._model_config.config.input]

    @property
    def input_dtypes(self) -> List[str]:
        """Return the data types of the input tensors of the model."""
        return [x.datatype for x in self._model_metadata.inputs]

    @property
    def output_names(self) -> List[str]:
        """Return the names of the output tensors of the model."""
        return [x.name for x in self._model_config.config.output]

    def make_request_data(
        self,
        images: np.ndarray
        ) -> Tuple[List[grpcclient.InferInput], List[grpcclient.InferRequestedOutput]]:
        """Create input and output data for an inference request.

        Args:
            images: Batch of images as a numpy array of shape
                (n, height, width, channels).

        Returns:
            A tuple of (inputs, outputs).
        """
        # Create inputs with the provided image data.
        inputs = [grpcclient.InferInput(
            self.input_names[0],
            images.shape,
            self.input_dtypes[0]
        )]
        inputs[0].set_data_from_numpy(images)

        # Create outputs container.
        outputs = []
        for name in self.output_names:
            outputs.append(grpcclient.InferRequestedOutput(name))
        return inputs, outputs

    def predict(self, images: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict on a batch of images.

        This method will package the input data, send it as a request to the
        inference server and return the results as numpy arrays.

        Args:
            images: Batch of images as a numpy array of shape
                (n, height, width, channels).

        Returns:
            A dictionary of numpy arrays with the inference results.
        """
        if self._client is None:
            self.connect()
        self._request_count += 1

        # Set up inference request.
        inputs, outputs = self.make_request_data(images)

        # 
        resp = self._client.infer(
            self.model_name,
            inputs,
            request_id=str(self._request_count),
            model_version=self.model_version,
            outputs=outputs
        )

        # Convert response to dictionary of arrays.
        response_data = {
            x.name: resp.as_numpy(x.name)
            for x in resp.get_response().outputs
        }
        return response_data


if __name__ == "__main__":
    from time import perf_counter

    video = sleap.load_video("../../tests/data/tracks/clip.mp4")
    
    client = InferenceClient()
    client.connect()

    dts = []
    for i in range((len(video))):
        img = video[i]

        t0 = perf_counter()
        preds = client.predict(img)
        dt = perf_counter() - t0
        dts.append(dt)
        # print(preds)
    dts = np.array(dts) * 1000

    print(f"Inference latency {img.shape}: {dts.mean():.2f} ms +- {dts.std():.1f} ms")

    import matplotlib.style
    import matplotlib as mpl
    mpl.rcParams["figure.facecolor"] = "w"
    mpl.rcParams["figure.dpi"] = 150
    mpl.rcParams["savefig.dpi"] = 600
    #mpl.rcParams["savefig.transparent"] = True
    # mpl.rcParams["savefig.bbox_inches"] = "tight"
    mpl.rcParams["font.size"] = 15
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial"]
    mpl.rcParams["axes.titlesize"] = "xx-large"  # medium, large, x-large, xx-large
    mpl.style.use("seaborn-deep")

    import matplotlib.pyplot as plt
    import seaborn as sns


    plt.figure(figsize=(6, 3))
    sns.histplot(dts, stat="probability")
    plt.xlabel("Inference latency (ms)")
    plt.ylabel("Probability")
    plt.xlim([0, 75])
    plt.show()

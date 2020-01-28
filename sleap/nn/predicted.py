import attr


@attr.s(auto_attribs=True)
class PredictedInstancePredictor:
    """
    Returns chunk of previously generated predictions in format of Predictor.
    """

    labels: "Labels"
    video_idx: int = 0

    def get_chunk(self, frame_inds):
        video = self.labels.videos[self.video_idx]

        # Return dict keyed to sample index (i.e., offset in frame_inds), value
        # is the list of instances for that frame.
        return {
            i: [
                inst
                for lf in self.labels.find(video=video, frame_idx=int(frame_idx))
                for inst in lf.instances
            ]
            for i, frame_idx in enumerate(frame_inds)
        }

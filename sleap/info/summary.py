import attr
import numpy as np


@attr.s(auto_attribs=True)
class Summary:
    labels: 'Labels'

    def get_point_count_series(self, video):
        series = dict()

        for lf in self.labels.find(video):
            val = sum(len(inst.points) for inst in lf if hasattr(inst, "score"))
            series[lf.frame_idx] = val
        return series

    def get_point_score_series(self, video, reduction="sum"):
        reduce_funct = dict(sum=sum, min=lambda x: min(x, default=0))[reduction]

        series = dict()

        for lf in self.labels.find(video):
            val = reduce_funct(point.score for inst in lf for point in inst.points if hasattr(inst, "score"))
            series[lf.frame_idx] = val
        return series

    def get_instance_score_series(self, video, reduction="sum"):
        reduce_funct = dict(sum=sum, min=lambda x: min(x, default=0))[reduction]

        series = dict()

        for lf in self.labels.find(video):
            val = reduce_funct(inst.score for inst in lf if hasattr(inst, "score"))
            series[lf.frame_idx] = val
        return series

    def get_point_displacement_series(self, video, reduction="sum"):
        reduce_funct = dict(sum=np.sum, mean=np.nanmean, max=np.max)[reduction]

        series = dict()

        last_lf = None
        for lf in self.labels.find(video):
            val = self._calculate_frame_velocity(lf, last_lf, reduce_funct)
            last_lf = lf
            if not np.isnan(val):
                series[lf.frame_idx] = val #len(lf.instances)
        return series

    @staticmethod
    def _calculate_frame_velocity(lf, last_lf, reduce_function):
        val = 0
        for inst in lf:
            if last_lf is not None:
                last_inst = last_lf.find(track=inst.track)
                if last_inst:
                    points_a = inst.visible_points_array
                    points_b = last_inst[0].visible_points_array
                    point_dist = np.linalg.norm(points_a - points_b, axis=1)
                    inst_dist = reduce_function(point_dist)
                    val += inst_dist if not np.isnan(inst_dist) else 0
        return val

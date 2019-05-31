import numpy as np
import itertools
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sleap.io.video import Video

class VideoFrameSuggestions:

	@classmethod
	def suggest(cls, video, params):
		method_functions = dict(strides=cls.strides, pca=cls.pca_cluster)
		method = params["method"]
		if method_functions.get(method, None) is not None:
			return method_functions[method](video=video, **params)

	@classmethod
	def pca_cluster(
			cls, video,
			clusters=5,
			per_cluster=5,
			sample_step=5,
			pca_components=50,
			interleave=True,
			*args, **kwargs):

		sample_count = video.frames//sample_step

		flat_stack = np.zeros((sample_count, video.height*video.width*video.channels))

		for i in range(sample_count):
			frame_idx = i * sample_step
			flat_stack[i] = video[frame_idx].flatten()

		pca = PCA(n_components=min(pca_components, sample_count))
		flat_small = pca.fit_transform(flat_stack)

		kmeans = KMeans(n_clusters=clusters)
		frame_labels = kmeans.fit_predict(flat_small)


		selected_by_cluster = []
		for i in range(clusters):
			bin,  = np.where(frame_labels==i)
			# print(f"{i}: {len(bin)}")
			samples_from_bin = np.random.choice(bin, min(len(bin), per_cluster), False)
			selected_by_cluster.append(samples_from_bin)

		# all_selected = list(all_selected)

		if interleave:
			# cycle clusters
			all_selected = itertools.chain.from_iterable(itertools.zip_longest(*selected_by_cluster))
			# remove Nones and convert back to list
			all_selected = list(filter(lambda x:x is not None, all_selected))
		else:
			all_selected = list(itertools.chain.from_iterable(selected_by_cluster))
			all_selected.sort()

		# convert sample index back into frame_idx
		all_selected = list(map(lambda x: int(x*sample_step), all_selected))

		# print(all_selected)
		return all_selected

	@classmethod
	def strides(
			cls, video, strides_per_video=20,
			*args, **kwargs):
	    suggestions = list(range(0, video.frames, video.frames//strides_per_video))
	    suggestions = suggestions[:strides_per_video]
	    return suggestions

if __name__ == "__main__":
	# load some images
	filename = "tests/data/videos/centered_pair_small.mp4"
	video = Video.from_filename(filename)

	print(VideoFrameSuggestions.suggest(video, dict(method="pca", interleave=False)))
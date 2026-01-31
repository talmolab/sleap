# Instance Size Distribution

*Case: You want to determine the optimal crop size for top-down models.*

When using top-down pose estimation pipelines, SLEAP crops around each detected instance before estimating pose. Choosing the right crop size is important:

- **Too small**: Parts of the animal may be cut off
- **Too large**: Wastes computation and may include other animals

The Instance Size Distribution widget helps you analyze your labeled data to choose an appropriate crop size.

## Accessing the Widget

From the GUI: **Analyze** → **Instance Size Distribution...**

## Understanding the Distribution

The widget displays a histogram of instance bounding box sizes across your labeled frames:

- **X-axis**: Bounding box size (in pixels)
- **Y-axis**: Number of instances

## Choosing a Crop Size

1. Open **Analyze** → **Instance Size Distribution...**
2. Review the histogram to see the range of instance sizes
3. Choose a crop size that covers the majority of instances (e.g., 95th percentile)
4. Consider adding padding (10-20%) for animals near frame edges

## Tips

- **Consistent animal sizes**: If your animals are similar sizes, the distribution will be tight and crop size selection is straightforward
- **Variable sizes**: If sizes vary significantly (e.g., adults and juveniles), consider using a larger crop size or filtering your training data
- **Multiple videos**: Check the distribution across all videos to ensure your crop size works for different recording conditions

## Using with Crop Size Visualization

After choosing a crop size, you can visualize it in the main view:

1. Open the training dialog (**Predict** → **Run Training...**)
2. The crop size overlay will show the crop region on your video
3. Scrub through frames to verify the crop captures the full animal

## Related

- [Configuring Models](https://nn.sleap.ai/latest/reference/models/) - Full details on crop size and other model parameters
- [Creating a Custom Training Profile](creating-a-custom-training-profile.md) - How to save your crop size settings

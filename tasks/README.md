# Processing Tasks

This directory holds configuration files for processing tasks.

A processing task is defined by:

- a detection model
- post-detection tasks
- tracked data
- whether to use a re-identification embedding

#### Post-Detection Task Options

- fine-grained classification with classification model (not implemented)
- dangerous goods detection (not implemented)
- company logo detection (not implemented)
- type of goods classification (not implemented)

#### Tracked Data Options

- vehicle class
- vehicle class confidence
- speed
- color
- entry direction
- exit direction
- dangerous goods info (not implemented)
- company logo info (not implemented)
- type of good info (not implemented)

#### Re-Identification Embedding

This option is useful if you know a video uses a camera angle prone to
a lot of occlusion. It may slow down processing speed, so there is a trade-off

### Example

An example task for processing a video may require the following:

1. Detecting 5 classes of vehicles
2. Fine-grained classification of a few of those classes after detection
3. Detecting dangerous goods symbols on vehicles

tracked data:

- class
- speed
- color
- dangerous goods info

re-identification: off

This config file would look like the following:

## TODO

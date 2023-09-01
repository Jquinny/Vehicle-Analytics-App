# Folder Structure

**Note:** Follow this directory structure if you don't want to redo the polygon
ROI and coordinate selection steps when processing videos from the same camera view

- The application can look for the json file containing the ROI and coordinate
  information in the folder structure below

```
videos
    |
    ├── <camera view 1>
    |       |
    |       ├── <video 1>
    |       .
    |       .
    |       ├── <video n>
    |       └── user_info.json
    ├── <camera view 2>
    |       |
    |       ├── <video 1>
    |       .
    |       .
    |       ├── <video n>
    |       └── user_info.json
```

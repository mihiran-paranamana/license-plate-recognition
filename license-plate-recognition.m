clc; clear all; close all;
numList = {};
% The foreground detector requires a certain number of video frames in
% order to initialize the Gaussian mixture model. This code uses the
% first 150 frames to initialize three Gaussian modes in the mixture model.
foregroundDetector = vision.ForegroundDetector('NumGaussians', 3, ...
    'NumTrainingFrames', 50, ...
    'MinimumBackgroundRatio', 0);

% Use the newly trained classifier to detect Licence Plates in the image.
detector = vision.CascadeObjectDetector('Train\LicencePlateDetector.xml','UseROI',true);   
detector.MergeThreshold = 20;

% Load the video using a video reader object.
videoReader = vision.VideoFileReader('Test\TestVd2.mp4');

% Create the webcam object.
% cam = webcam();

for i = 1:180
    
    % read the next video frame
    frame = step(videoReader);

    % Get the next frame.
    % frame = snapshot(cam);
    
    foreground = step(foregroundDetector, frame);
end

% Remove small objects from binary image
filteredForeground1 = bwareaopen(foreground, 100);
    
% Use morphological close to fill gaps in the detected objects.
se = strel('disk', 12);
filteredForeground2 = imclose(filteredForeground1, se);

% figure; imshow(filteredForeground2); title('Filtered Foreground');

% Next, we find bounding boxes of each connected component corresponding to
% a moving vehicle by using vision.BlobAnalysis object. The object further
% filters the detected foreground by rejecting blobs which contain fewer
% than 700 pixels.
blobAnalysis = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
    'AreaOutputPort', false, 'CentroidOutputPort', false, ...
    'MinimumBlobArea', 700);

% Process the Rest of Video Frames
videoPlayer = vision.VideoPlayer('Name', 'Detected Cars');
videoPlayer.Position(3:4) = [650,400];  % window size: [width, height]

runLoop = true;
while runLoop & ~isDone(videoReader)
    
    % read the next video frame
    frame = step(videoReader);
    
    % Get the next frame.
    % frame = snapshot(cam);
    
    % Detect the foreground in the current video frame
    foreground = step(foregroundDetector, frame);
    
    % Remove small objects from binary image
    filteredForeground1 = bwareaopen(foreground, 100);
    
    % Use morphological close to fill gaps in the detected objects.
    se = strel('disk', 12);
    filteredForeground2 = imclose(filteredForeground1, se);
    
    % Detect the connected components with the specified minimum area, and
    % compute their bounding boxes
    bboxes = step(blobAnalysis, filteredForeground2);
    
    result = frame;
    
    for i = 1:size(bboxes,1)
        
        % Detect interest points in such bounding boxes.
        points = detectMinEigenFeatures(rgb2gray(result),'ROI',bboxes(i,:));
        numPts = size(points.Location,1);
        
        % Set Threshold values to detect moving objects.
        if numPts < 340 | 380 < numPts
            bboxes(i,:) = zeros(1,4);
        else
            % Detect Licence Plates.
            bbox = step(detector,result,bboxes(i,:));
            if ~isempty(bbox)
                result = insertShape(result, 'Rectangle', bbox, 'Color', 'Red', 'LineWidth', 5);
                plate = ocr(frame, bbox);
                if ~isempty(plate)
                    for j = 1:size(plate,1)
                        if ~isempty(plate(j,1).Text)
                            numList{end+1} = plate(j,1).Text;
                        end
                    end
                end
            end
        end  
    end    
    
    % Draw bounding boxes around the detected cars
    result = insertShape(result, 'Rectangle', bboxes, 'Color', 'green', 'LineWidth',5);
        
    % display the results
    step(videoPlayer, result); % fliplr()
    
    % Checks if videoPlayer is still open
    runLoop = isOpen(videoPlayer);
end

% close the video file
release(videoReader);

% clear the camera object
% clear cam

disp(numList);

import numpy as np
import cv2
import copy
import argparse


def argsParser():
    # construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required=True,
		help="path to input video")
	ap.add_argument("-o", "--output", required=True,
		help="path to output video")
	args = vars(ap.parse_args())

	return args


def motion_heatmap(args):
    cap = cv2.VideoCapture(args["input"])
    
    # pip install opencv-contrib-python
    # substract background
    fgbg = cv2.createBackgroundSubtractorMOG2()
    writer = cv2.VideoWriter(args["output"],
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                int(cap.get(5)),
                                (int(cap.get(3)), int(cap.get(4))))

    first_iteration_indicator = 1
    while True:
        # read the next frame from the file
        ret, frame = cap.read()

        # if the frame was not grabbed, then we have reached the end of the stream
        if not ret:
            break
        '''
        Following are the reasons to have if statement:
            -in the first run there is no previous frame, so this accounts for that
            -the first frame is saved to be used for the overlay after the accumulation has occurred
            -the height and width of the video are used to create an empty image for accumulation (accum_image)
        '''
        if (first_iteration_indicator == 1):
            first_frame = copy.deepcopy(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape[:2]
            accum_image = np.zeros((height, width), np.uint8)
            first_iteration_indicator = 0
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale

            fgmask = fgbg.apply(gray)  # remove the background

            # apply a binary threshold only keeping pixels above thresh and setting the result to maxValue.  If you want
            # motion to be picked up more, increase the value of maxValue.  To pick up the least amount of motion over time, set maxValue = 1
            thresh = 2
            maxValue = 2
            ret, th1 = cv2.threshold(fgmask, thresh, maxValue, cv2.THRESH_BINARY)
            
            # add to the accumulated image
            accum_image = cv2.add(accum_image, th1)
            
            # apply a color map
            color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_TURBO)
            
            # overlay the color mapped image to the first frame
            result_overlay = cv2.addWeighted(frame, 0.7, color_image, 0.7, 0)

            # save the final overlay image
            writer.write(result_overlay)

    # cleanup
    writer.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = argsParser()
    motion_heatmap(args)

if __name__=='__main__':
    main()
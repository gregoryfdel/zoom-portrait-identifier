import cv2
import pytesseract
import numpy as np
import re

#windows specific requirement, comment out if not on windows
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

#List of filenames to analyze
fileNameList = ["zoom_call"]

#This script does not create any of these folders, one must do it themselves
#Root folder of images
workingD = "<working_dir>\\example\\"
#Output folder 
croppedDirPath = workingD + "output\\"

#If you do not want to generate these images, 
#replace with appropiate directory with None
#unnamedPath = workingD + "unnamed\\"
#namesPath = workingD + "names\\"
unnamedPath = None
namesPath = None


#Parameters of script
#The first step, which identfies the borders
#of each portrait. Try to capture the greyscale
#color of the border color.
#thresImage = (26, 28)
thresImage = (0,15)
#Invert the image if needed to find borders
invertImage = False

#Name finding, use these to find the name 
#in the portrait
#Name extents
#Search for the name in a specific rectangle on the portrait
nameExtent = [(0,298),(308,380)]
#Make the text color black, with everything else white
#nameFontColorRange = (0, 85)
nameFontColorRange = (235, 255)
#The last cleaning step is to remove any black blotches
#with a bigger area than this
#cleanupParameter = 170
cleanupParameter = 500

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

already_found_names = [""]
pattern = re.compile('[\W_ ]+', re.UNICODE)


for zoomCallScreen in fileNameList:
	croppedDirPath = workingD + "output\\"
	filename = workingD + zoomCallScreen + ".png"
	print("=================")
	print(filename)
	print("=================")
	# Read the image
	img = cv2.imread(filename, 0)
	
	# Thresholding the image
	#img_bin = cv2.inRange(img, 26, 28)
	img_bin = cv2.inRange(img, 0, 15)
	if invertImage:
		img_bin = 255-img_bin
    
	# Defining a kernel length
	kernel_length = np.array(img).shape[1]//80
	
	# A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
	verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))# A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
	hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))# A kernel of (3 X 3) ones.
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	# Morphological operation to detect vertical lines from an image
	img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
	verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
	img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
	horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
	
	# Weighting parameters, this will decide the quantity of an image to be added to make a new image.
	alpha = 0.5
	beta = 1.0 - alpha# This function helps to add two image with specific weight parameter to get a third image as summation of two image.
	img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
	img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
	(thresh, img_final_bin) = cv2.threshold(img_final_bin, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	
	# Find contours for image, which will detect all the boxes
	contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)# Sort all the contours by top to bottom.
	(contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")
	
	#Now crop the big image to each persons portrait
	idx = 0
	color_img = cv2.imread(filename, -1)
	for c in contours:
		# Returns the location and width,height for every contour
		x, y, w, h = cv2.boundingRect(c)
		idx += 1
		new_color_img = color_img[y:y+h, x:x+w]
		if unnamedPath is not None:
			cv2.imwrite(unnamedPath+str(idx)+ '.png', new_color_img)# If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
		#Try to figure out the name of this person
		new_img = img[y:y+h, x:x+w]
		name_img = new_img[nameExtent[0][1]:nameExtent[1][1],nameExtent[0][0]:nameExtent[1][0]]
		if len(name_img) < 1:
			continue
		name_bin = cv2.inRange(name_img, nameFontColorRange[0], nameFontColorRange[1])
		#We do not want any weird blobs which will interfere with tesseract
		name_contours, name_hierarchy = cv2.findContours(name_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)# Sort all the contours by top to bottom.
		name_bin = 255 - name_bin
		if len(name_contours) > 0:
			for ci,c2 in enumerate(name_contours):
				xl, yl, wl, hl = cv2.boundingRect(c2)
				if (wl*hl > cleanupParameter):
					cimg = np.zeros_like(name_bin)
					cv2.drawContours(cimg, name_contours, ci, color=255, thickness=-1)
					pts = np.where(cimg == 255)
					name_bin[pts[0], pts[1]] = 255
		if namesPath is not None:
			cv2.imwrite(namesPath+str(idx)+ '.png', name_bin)
		text = pytesseract.image_to_string(name_bin)
		#No strange charaters, only ascii
		text = text.encode("ascii", errors="ignore").decode()
		text = pattern.sub('', text)
		text = text.strip()
		if text in already_found_names:
			continue
		already_found_names.append(text)
		print(idx)
		print(text)
		cv2.imwrite(croppedDirPath+text+ '.png', new_color_img)# If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
import cv2

video_counter = 1
num_of_vids = 30

while video_counter <= num_of_vids:
	vidcap = cv2.VideoCapture('C:\\Users\\nbloc\\Desktop\\Data\\' + str(video_counter) + '\\video' + str(video_counter) + '.mp4')
	success,image = vidcap.read()
	count = 0
	while success:
		image = image[5:1035, 545:1375]
		cv2.imwrite('C:\\Users\\nbloc\\Desktop\\Data\\' + str(video_counter) + "\\frames" + str(video_counter) +"\\frame" + str(count) + ".jpg", image)      
		success,image = vidcap.read()
		count += 1
    
	print("Finished parsing video" + str(video_counter))
  	#dim = (100, 56)
  	#resized = cv2.resize(image, dim)
  	#cv2.imwrite("C:\\Users\\nbloc\\Desktop\\framesResized\\frame%d.jpg" % count, resized)     # save frame as JPEG file 
	video_counter += 1

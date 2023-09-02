import cv2

img = cv2.imread('../DATA/00-puppy.jpg')

while True:
    cv2.imshow("Puppy", img)
    
    # If we've waited 1 ms and we've pressed esc
    if cv2.waitKey(1) & 0xFF == 27:
        break
        
cv2.destroyAllWindows()

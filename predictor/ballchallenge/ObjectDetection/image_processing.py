import cv2

class NoBoxDetectedError(Exception):
    def __init__(self, message="The box was not properly detected"):
        super().__init__(message)


def process_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(cleaned_image, 100, 200)
    #Sandsack muss voll geschlossen sein

    contours_numpy, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x = None
    y = None
    w = None
    h = None
    for contour in contours_numpy:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
    if x is None or y is None or w is None or h is None:
        raise NoBoxDetectedError()
    return x, y, w, h




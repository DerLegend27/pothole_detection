import numpy as np
import tensorflow as tf
import cv2
import os

from PIL import Image

from utils import label_map_util
from utils import visualization_utils as vis_util

# --> Globale Variablen <--

# Dateipfad zur Modelldatei
MODEL =  r"model/pothole_resnet.pb"

# Dateipfad zu den Labels
LABELS = r"model/LabelMap.txt"
CLASSES = 8

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def preparation():
    counter = 1
    for file in os.listdir("images/"):
        img = cv2.imread("images/"+str(file))

        # Save Original Image
        cv2.imwrite(f"images/{counter}.jpg", img)

        height, width, _ = img.shape

        box = np.array([[0, 0], [width, 0], [width, height//3], [0, height//3]])
        color = [0, 0, 0]

        cv2.fillConvexPoly(img, box, color)
        
        #cv2.namedWindow('finalImg', cv2.WINDOW_NORMAL)
        #cv2.imshow("finalImg",img)                     
        #cv2.waitKey(0)

        cv2.imwrite(f"images/{counter}_prepared.jpg", img)
        os.remove("images/"+str(file))
        counter+=1

def detection():
    # Detection Graph erstellen
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v2.io.gfile.GFile(MODEL, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)

            tf.import_graph_def(od_graph_def, name="")

            label_map = label_map_util.load_labelmap(LABELS)
            categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=CLASSES, use_display_name=True)
            category_index = label_map_util.create_category_index(categories)

            counter = 1
            for file in os.listdir("images/"):
                if "prepared" not in str(file):
                    with detection_graph.as_default():
                        with tf.compat.v1.Session(graph=detection_graph) as sess:
                            # Erstellen von Tensoren für den Detection Graph
                            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                            # Gefundenes Schlagloch oder Straßenschäden -> Box-Koordinaten
                            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                            # Information über den gefunden Schaden wie z.B. prozentuale Sicherheit der Erkennung oder das zugehörige Label
                            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                            image = Image.open("images/"+str(file))
                            #image_copy = Image.open(f"images/{counter}.jpg")

                            # Image in Numpy Array [1,...,0] umgewandelt
                            image_np = load_image_into_numpy_array(image)
                            #image_np_copy = load_image_into_numpy_array(image_copy)

                            # Image transformation um folgendes Format zu erhalten: [1, None, None, 3]
                            image_np_expanded = np.expand_dims(image_np, axis=0)

                            # Der abschließende Prozess und die eigentliche Rechenarbeit
                            (boxes, scores, classes, num) = sess.run(
                                [detection_boxes, detection_scores, detection_classes, num_detections],
                                feed_dict={image_tensor: image_np_expanded})

                            # Schaden auf dem resultierenden Bild anzeigen lassen
                            vis_util.visualize_boxes_and_labels_on_image_array(
                                image_np,
                                np.squeeze(boxes),
                                np.squeeze(classes).astype(np.int32),
                                np.squeeze(scores),
                                category_index,
                                min_score_thresh=0.2,
                                use_normalized_coordinates=True,
                                line_thickness=8)

                            cv2.imwrite(f"result_images/{counter}_result.jpg", image_np)
                            #cv2.imshow("Detection", image_np)
                            #cv2.waitKey(0)

                            counter+=1  

if __name__ == "__main__":

    if os.listdir("images/") :
        #preparation()
        detection()
    else:
        print("Es sind keine Bilder im Ordner vorhanden!")
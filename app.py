import face_recognition
import cv2
import numpy as np
from flask import Flask,render_template,Response

app=Flask(__name__)
camera=cv2.VideoCapture(0)

# Load a sample pictures (2)
Manoswita_image = face_recognition.load_image_file("Manoswita/Manoswita.jpg")
Manoswita_face_encoding = face_recognition.face_encodings(Manoswita_image)[0]

Soumita_image = face_recognition.load_image_file("Soumita/Soumita.jpg")
Soumita_face_encoding = face_recognition.face_encodings(Soumita_image)[0]

# Create arrays 
known_face_encodings = [
    Manoswita_face_encoding,
    Soumita_face_encoding
]
known_face_names = [
    "Manoswita",
    "Soumita"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def generate_frames():
    while True: 
        success,frame=camera.read()
        if not success:
            break
        else:
            # Resize frame 
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # BGR color to RGB color 
            rgb_small_frame = small_frame[:, :, ::-1]
            
            # For face in current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    print (1)
                else:
                    print(0)
                face_names.append(name)
                


            # Results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale up
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('Index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=False)
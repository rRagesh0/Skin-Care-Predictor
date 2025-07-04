from flask import Flask, render_template, request
import os
from model.predict import predict_skin_type_condition  # Import your prediction model
from recommender.recommender import recommender  # Import your recommender function

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded_images'

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if a file is part of the request
        if 'file' not in request.files:
            print("No file uploaded")
            return "No file uploaded"

        file = request.files['file']

        # If no file is selected
        if file.filename == '':
            print("No file selected")
            return "No file selected"

        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        try:
            file.save(file_path)
            print(f"File saved at: {file_path}")
        except Exception as e:
            print(f"Error saving file: {e}")
            return "Error saving file"

        # Predict skin type and condition
        try:
            skin_type, skin_condition = predict_skin_type_condition(file_path)
            print(f"Predicted Skin Type: {skin_type}, Skin Condition: {skin_condition}")
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "Error in prediction"

        # Get recommendations
        try:
            recommendations = recommender(skin_type, skin_condition)
            print(f"Recommendations: {recommendations}")
        except Exception as e:
            print(f"Error in recommendation: {e}")
            return "Error in recommendation"

        # Render results
        if recommendations:
            return render_template(
                "results.html", 
                skin_type=skin_type, 
                skin_condition=skin_condition, 
                recommendations=recommendations
            )
        else:
            print("No recommendations found")
            return render_template(
                "results.html", 
                skin_type=skin_type, 
                skin_condition=skin_condition, 
                recommendations=[]
            )

    return render_template("index.html")



if __name__ == "__main__":
    app.run(debug=True)

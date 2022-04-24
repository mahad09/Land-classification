def single_prediction(validation_data):
  trained_model = keras.models.load_model('land_classifier.h5')
  image_path = TRAINING_DATA_DIR + 'agricultural/' + 'agricultural03.png'
  img = cv2.imread(image_path)
  img = cv2.resize(img, (128, 128))
  img = np.array(img)
  img = img.reshape(1, 128, 128, 3)
  img *= 255
  predict = np.argmax(trained_model.predict(img))

  for key, value in validation_data.class_indices.items():
    if value == int(predict):
        print(key)

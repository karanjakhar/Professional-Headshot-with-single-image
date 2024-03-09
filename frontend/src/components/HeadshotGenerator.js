import React, { useState } from 'react';

function HeadshotGenerator() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [resultImage, setResultImage] = useState(null);

  const handleImageChange = (e) => {
    setSelectedImage(e.target.files[0]);
  };

  const handleUploadImage = async () => {
    const formData = new FormData();
    formData.append('file', selectedImage);
    try {
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      console.log(response);
      console.log(data.uid)
      const result_url = 'http://localhost:8000/get_result_image/' + data.uid;
      setResultImage(result_url);
    } catch (error) {
      console.error('Error uploading image:', error);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <h1 className="text-3xl font-semibold mb-8">Professional Headshot Generator</h1>
      <div className="w-full max-w-lg bg-white p-8 rounded-lg shadow-lg mb-8">
        <input
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          className="mb-4"
          name="file"
        />
        {selectedImage && (
          <img src={URL.createObjectURL(selectedImage)} alt="Selected" className="mb-4 rounded-lg" />
        )}
        <button
          onClick={handleUploadImage}
          className="bg-blue-500 text-white py-2 px-4 rounded-lg hover:bg-blue-600"
        >
          Upload Image
        </button>
        {resultImage && (
          <div>
            <h2 className="text-xl font-semibold mt-4">Result Image:</h2>
            <img src={resultImage} alt="Result" className="mt-2 rounded-lg" />
          </div>
        )}
      </div>
    </div>
  );
}

export default HeadshotGenerator;

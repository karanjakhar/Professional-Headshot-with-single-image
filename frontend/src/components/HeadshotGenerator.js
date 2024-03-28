import React, { useState } from 'react';

function HeadshotGenerator() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    setSelectedImage(e.target.files[0]);
  };

  const handleUploadImage = async () => {
    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedImage);
    try {
      const response = await fetch('https://karanjakhar--professional-headshot-single-image-app.modal.run/upload', {
        method: 'POST',
        body: formData,
      });
      const blob = await response.blob();
    const url = URL.createObjectURL(blob);
  
      setResultImage(url);
    } catch (error) {
      console.error('Error uploading image:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center mx-6 bg-white">
      <h1 className="text-3xl font-semibold mb-8">Select and upload</h1>
      <div className=" bg-white p-8 rounded-lg shadow-lg mb-8">
        <input
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          className="mb-4"
          name="file"
        />
        {selectedImage && (
          <img src={URL.createObjectURL(selectedImage)} alt="Selected" width={200} className="mb-4 rounded-lg" />
        )}
        <button
          onClick={handleUploadImage}
          className="bg-red-500 text-white py-2 px-4 rounded-lg hover:bg-red-400"
          disabled={!selectedImage || loading}
        >
          {loading ? 'Uploading...' : 'Upload Image'}
        </button>
        {resultImage && (
          <div>
            <h2 className="text-xl font-semibold mt-4">Result Image:</h2>
            <img src={resultImage} alt="Result" width={200} className="mt-2 rounded-lg" />
          </div>
        )}
      </div>
    </div>
  );
}

export default HeadshotGenerator;

import './App.css';
import HeadshotGenerator from './components/HeadshotGenerator';
import Navbar from './components/Navbar';
import { BrowserRouter } from 'react-router-dom';
import Footer from './components/Footer';

import Hero from './components/Hero';
import Features from './components/Features';
import Testimonial from './components/Testimonial';
import CallToAction from './components/CallToAction';
function App() {
  return (
    <div>
      <BrowserRouter>
      <Navbar />
      <Hero />
      <Features />
      <Testimonial />
      <CallToAction />
      <Footer/>
      </BrowserRouter>
   
    </div>
  );
}

export default App;

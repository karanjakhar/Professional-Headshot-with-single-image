import { Link } from 'react-router-dom';
import HeadshotGenerator from './HeadshotGenerator';

const Hero = () => {
  return (
    <section id='hero'>
      {/* Flex Container */}
      <div className='container flex flex-col-reverse items-center px-6 mx-auto mt-10 space-y-0 md:space-y-0 md:flex-row'>
        {/* Left Item */}
        <div className='flex flex-col mb-32 space-y-12 md:w-1/2'>
          <h1 className='max-w-md text-4xl font-bold text-center md:text-5xl md:text-left'>
            Professional Headshots Generator
          </h1>
          <p className='max-w-sm text-center text-blue-950 md:text-left'>
           PHS creates high quality Professional headshots using just a <b>Single Image</b>.
          </p>
          <div className='flex justify-center md:justify-start'>
            <Link
              to='#'
              className='p-3 px-6 pt-2 text-white bg-red-500 rounded-full baseline hover:bg-red-400'
            >
              Get Started
            </Link>
          </div>
        </div>
        {/* Image */}
        <div className='md:w-1/2'>
          {/* <img src={illustrationIntro} alt='' /> */}
          <HeadshotGenerator />
        </div>
      </div>
    </section>
  );
};

export default Hero;
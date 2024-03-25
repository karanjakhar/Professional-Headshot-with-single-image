import React from 'react';

const Features = () => {
  return (
    <section id='features'>
      {/* Flex Container */}
      <div className='container flex flex-col px-4 mx-auto mt-10 space-y-12 md:space-y-0 md:flex-row'>
        {/* What's Different */}
        <div className='flex flex-col space-y-12 md:w-1/2'>
          <h2 className='max-w-md text-4xl font-bold text-center md:text-left'>
            What's different about PHS?
          </h2>
          <p className='max-w-sm text-center text-darkGrayishBlue md:text-left'>
            PHS only need a single image to create high quality Professional headshots. It's very fast. Takes only a few seconds.
          </p>
        </div>

         
        </div>
    </section>
  );
};

export default Features;
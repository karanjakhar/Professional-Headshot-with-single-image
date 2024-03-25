import { Link } from 'react-router-dom';

import avatarAnisha from '../assets/images/avatar-anisha.png';
import avatarAli from '../assets/images/avatar-ali.png';
import avatarRichard from '../assets/images/avatar-richard.png';

const Testimonial = () => {
  return (
    <section id='testimonials'>
      {/* Container to heading and testm blocks */}
      <div className='max-w-6xl px-5 mx-auto mt-32 text-center'>
        {/* Heading */}
        <h2 className='text-4xl font-bold text-center'>
          What Our Customers Say
        </h2>
        {/* Testimonials Container */}
        <div className='flex flex-col mt-24 md:flex-row md:space-x-6'>
          {/* Testimonial 1 */}
          <div className='flex flex-col items-center p-6 space-y-6 rounded-lg bg-gray-200 md:w-1/3'>
            <img src={avatarAnisha} className='w-16 -mt-14' alt='' />
            <h5 className='text-lg font-bold'>Anisha Li</h5>
            <p className='text-sm text-slate-600'>
              “Best photo generator!!”
            </p>
          </div>

          {/* Testimonial 2 */}
          <div className='hidden flex-col items-center p-6 space-y-6 rounded-lg bg-gray-200 md:flex md:w-1/3'>
            <img src={avatarAli} className='w-16 -mt-14' alt='' />
            <h5 className='text-lg font-bold'>Ali Bravo</h5>
            <p className='text-sm text-slate-600'>
              “It's very fast and high quality.”
            </p>
          </div>

          {/* Testimonial 3 */}
          <div className='hidden flex-col items-center p-6 space-y-6 rounded-lg bg-gray-200 md:flex md:w-1/3'>
            <img src={avatarRichard} className='w-16 -mt-14' alt='' />
            <h5 className='text-lg font-bold'>Richard Watts</h5>
            <p className='text-sm  text-slate-600'>
              “Saves a lot of time and very easy to use.”
            </p>
          </div>
        </div>
        {/* Button */}
        <div className='my-16'>
          <Link
            to='#'
            className='p-3 px-6 pt-2 text-white bg-red-500 rounded-full baseline hover:bg-red-400'
          >
            Get Started
          </Link>
        </div>
      </div>
    </section>
  );
};

export default Testimonial;
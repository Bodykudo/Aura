import { createLazyFileRoute } from '@tanstack/react-router';

function Noise() {
  return (
    <>
      <p>NOISE</p>
    </>
  );
}

export const Route = createLazyFileRoute('/noise')({
  component: Noise
});

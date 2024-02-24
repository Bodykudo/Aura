import { createLazyFileRoute } from '@tanstack/react-router';

function Thresholding() {
  return (
    <>
      <p>Thresholding</p>
    </>
  );
}

export const Route = createLazyFileRoute('/thresholding')({
  component: Thresholding
});

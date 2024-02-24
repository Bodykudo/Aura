import { createLazyFileRoute } from '@tanstack/react-router';

function Histogram() {
  return (
    <>
      <p>Histogram</p>
    </>
  );
}

export const Route = createLazyFileRoute('/histogram')({
  component: Histogram
});

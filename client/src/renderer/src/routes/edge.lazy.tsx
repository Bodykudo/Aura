import { createLazyFileRoute } from '@tanstack/react-router';

function Edge() {
  return (
    <>
      <p>Edge</p>
    </>
  );
}

export const Route = createLazyFileRoute('/edge')({
  component: Edge
});

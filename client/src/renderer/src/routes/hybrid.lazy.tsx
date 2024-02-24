import { createLazyFileRoute } from '@tanstack/react-router';

function Hybrid() {
  return (
    <>
      <p>Hybrid</p>
    </>
  );
}

export const Route = createLazyFileRoute('/hybrid')({
  component: Hybrid
});

import { createRootRoute, Outlet } from '@tanstack/react-router';
import { TanStackRouterDevtools } from '@tanstack/router-devtools';

import Sidebar from '@renderer/components/Sidebar';

export const Route = createRootRoute({
  component: () => (
    <div className="h-full relative">
      <Sidebar />

      <main className="md:pl-72">
        <Outlet />
      </main>
      <TanStackRouterDevtools />
    </div>
  )
});

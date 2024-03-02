import { createRootRoute, Outlet } from '@tanstack/react-router';
import { TanStackRouterDevtools } from '@tanstack/router-devtools';
import { ThemeProvider } from 'next-themes';

import Sidebar from '@renderer/components/Sidebar';
import { Toaster } from '@renderer/components/ui/toaster';
import ModeToggle from '@renderer/components/ModeToggle';
import MobileSidebar from '@renderer/components/MobileSidebar';

export const Route = createRootRoute({
  component: () => (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
      <div className="h-full relative">
        <Toaster />
        <div className="hidden h-full md:flex md:w-72 md:flex-col md:fixed md:inset-y-0 bg-gray-900 overflow-y-auto">
          <Sidebar />
        </div>

        <main className="md:pl-72">
          <div className="flex justify-between p-4">
            <MobileSidebar />
            <ModeToggle />
          </div>
          <Outlet />
        </main>
      </div>
    </ThemeProvider>
  )
});

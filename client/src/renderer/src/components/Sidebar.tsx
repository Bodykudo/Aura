import { Link, useRouterState } from '@tanstack/react-router';

import { routes } from '@renderer/lib/constants';
import { cn } from '@renderer/lib/utils';

import logo from '@renderer/assets/logo.png';

export default function Sidebar() {
  const router = useRouterState();

  return (
    <div className="flex flex-col space-y-4 py-4 h-full bg-[#111827] text-white">
      <div className="px-3 py-2 flex-1">
        <Link to="/" className="flex mt-4 mb-8">
          <div className="h-12 mx-auto">
            <img src={logo} className="mx-auto h-full" alt="logo" />
          </div>
        </Link>

        <div className="space-y-1">
          {routes.map((route) => (
            <Link
              to={route.href}
              key={route.href}
              className={cn(
                'text-sm group flex p-3 w-full justify-start font-medium cursor-pointer hover:text-white hover:bg-white/10 transition-all rounded-lg text-zinc-400',
                router.location.pathname === route.href && 'bg-white/10'
              )}
            >
              <div className="flex items-center flex-1">
                <route.icon className={cn('h-5 w-5 mr-3', route.color)} />
                {route.label}
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}

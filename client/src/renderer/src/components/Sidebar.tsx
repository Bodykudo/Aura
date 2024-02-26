import { routes } from '@renderer/lib/constants';
import { cn } from '@renderer/lib/utils';
import { Link } from '@tanstack/react-router';

export default function Sidebar() {
  return (
    <div className="hidden h-full md:flex md:w-72 md:flex-col md:fixed md:inset-y-0 bg-gray-900 overflow-y-auto">
      <div className="flex flex-col space-y-4 py-4 h-full bg-[#111827] text-white">
        <div className="px-3 py-2 flex-1">
          <Link to="/" className="flex pl-3 mb-14">
            <div className="relative h-12 w-36 sm:h-16 sm:w-48 mx-auto">
              LOGO IS HERE
              {/* <img src="/logo-dark.png" alt="logo" fill /> */}
            </div>
          </Link>

          <div className="space-y-1">
            {routes
              .filter((route) => !route.comingSoon)
              .map((route) => (
                <Link
                  to={route.href}
                  key={route.href}
                  className="text-sm group flex p-3 w-full justify-start font-medium cursor-pointer hover:text-white hover:bg-white/10 transition-all rounded-lg text-zinc-400"
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
    </div>
  );
}

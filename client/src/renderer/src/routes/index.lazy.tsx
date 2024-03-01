import { Card } from '@renderer/components/ui/card';
import { routes } from '@renderer/lib/constants';
import { cn } from '@renderer/lib/utils';
import { createLazyFileRoute, useNavigate } from '@tanstack/react-router';
import { ArrowRight } from 'lucide-react';

function Index() {
  const navigate = useNavigate({ from: '/' });

  return (
    <div>
      <div className="mb-8 space-y-4">
        <h2 className="text-2xl md:text-4xl font-bold text-center">
          Aura - Image Processing Playground
        </h2>
        <p className="text-muted-foreground font-light text-sm md:text-lg text-center">
          A playground for image processing algorithms, filters, and more. Explore and learn about
          the world of image processing.
        </p>
      </div>
      <div className="px-4 md:px-20 lg:px-32 space-y-4">
        {routes.map((route) => (
          <Card
            key={route.href}
            onClick={() => navigate({ to: route.href })}
            className={cn(
              'p-4 border-black/5 dark:border-gray-800 flex items-center justify-between hover:shadow-md transition-all cursor-pointer',
              route.comingSoon && 'opacity-50 pointer-events-none'
            )}
          >
            <div className="flex items-center gap-x-4">
              <div className={cn('p-2 w-fit rounded-md', route.bgColor)}>
                <route.icon className={cn('w-8 h-8', route.color)} />
              </div>
              <div className="font-semibold flex items-center gap-8">
                {route.label}
                {route.comingSoon && <span className="text-xs ">(Coming Soon)</span>}
              </div>
            </div>
            <ArrowRight className="w-5 h-5" />
          </Card>
        ))}
      </div>
    </div>
  );
}

export const Route = createLazyFileRoute('/')({
  component: Index
});

import { createFileRoute } from '@tanstack/react-router';

// Import Routes
import { Route as rootRoute } from './routes/__root';

// Create Virtual Routes
const IndexLazyImport = createFileRoute('/')();
const NoiseLazyImport = createFileRoute('/noise')();
const FiltersLazyImport = createFileRoute('/filters')();
const EdgeLazyImport = createFileRoute('/edge')();
const HistogramLazyImport = createFileRoute('/histogram')();
const ThresholdingLazyImport = createFileRoute('/thresholding')();
const HybridLazyImport = createFileRoute('/hybrid')();
const HoughLazyImport = createFileRoute('/hough')();
const ContoursLazyImport = createFileRoute('/contours')();
const CornersLazyImport = createFileRoute('/corners')();
const MatchingLazyImport = createFileRoute('/matching')();
const SiftLazyImport = createFileRoute('/sift')();
const SegmentationLazyImport = createFileRoute('/segmentation')();
const FaceLazyImport = createFileRoute('/face')();

// Create/Update Routes
const IndexLazyRoute = IndexLazyImport.update({
  path: '/',
  getParentRoute: () => rootRoute
} as any).lazy(() => import('./routes/index.lazy').then((d) => d.Route));

const NoiseLazyRoute = NoiseLazyImport.update({
  path: '/noise',
  getParentRoute: () => rootRoute
} as any).lazy(() => import('./routes/noise.lazy').then((d) => d.Route));

const FiltersLazyRoute = FiltersLazyImport.update({
  path: '/filters',
  getParentRoute: () => rootRoute
} as any).lazy(() => import('./routes/filters.lazy').then((d) => d.Route));

const EdgeLazyRoute = EdgeLazyImport.update({
  path: '/edge',
  getParentRoute: () => rootRoute
} as any).lazy(() => import('./routes/edge.lazy').then((d) => d.Route));

const HistogramLazyRoute = HistogramLazyImport.update({
  path: '/histogram',
  getParentRoute: () => rootRoute
} as any).lazy(() => import('./routes/histogram.lazy').then((d) => d.Route));

const ThresholdingLazyRoute = ThresholdingLazyImport.update({
  path: '/thresholding',
  getParentRoute: () => rootRoute
} as any).lazy(() => import('./routes/thresholding.lazy').then((d) => d.Route));

const HybridLazyRoute = HybridLazyImport.update({
  path: '/hybrid',
  getParentRoute: () => rootRoute
} as any).lazy(() => import('./routes/hybrid.lazy').then((d) => d.Route));

const HoughLazyRoute = HoughLazyImport.update({
  path: '/hough',
  getParentRoute: () => rootRoute
} as any).lazy(() => import('./routes/hough.lazy').then((d) => d.Route));

const ContoursLazyRoute = ContoursLazyImport.update({
  path: '/contours',
  getParentRoute: () => rootRoute
} as any).lazy(() => import('./routes/contours.lazy').then((d) => d.Route));

const CornersLazyRoute = CornersLazyImport.update({
  path: '/corners',
  getParentRoute: () => rootRoute
} as any).lazy(() => import('./routes/corners.lazy').then((d) => d.Route));

const MatchingLazyRoute = MatchingLazyImport.update({
  path: '/matching',
  getParentRoute: () => rootRoute
} as any).lazy(() => import('./routes/matching.lazy').then((d) => d.Route));

const SiftLazyRoute = SiftLazyImport.update({
  path: '/sift',
  getParentRoute: () => rootRoute
} as any).lazy(() => import('./routes/sift.lazy').then((d) => d.Route));

const SegmentationLazyRoute = SegmentationLazyImport.update({
  path: '/segmentation',
  getParentRoute: () => rootRoute
} as any).lazy(() => import('./routes/segmentation.lazy').then((d) => d.Route));

const FaceLazyRoute = FaceLazyImport.update({
  path: '/face',
  getParentRoute: () => rootRoute
} as any).lazy(() => import('./routes/face.lazy').then((d) => d.Route));

// Populate the FileRoutesByPath interface
declare module '@tanstack/react-router' {
  interface FileRoutesByPath {
    '/': {
      preLoaderRoute: typeof IndexLazyImport;
      parentRoute: typeof rootRoute;
    };
    '/noise': {
      preLoaderRoute: typeof NoiseLazyImport;
      parentRoute: typeof rootRoute;
    };
    '/filters': {
      preLoaderRoute: typeof FiltersLazyImport;
      parentRoute: typeof rootRoute;
    };
    '/edge': {
      preLoaderRoute: typeof EdgeLazyImport;
      parentRoute: typeof rootRoute;
    };
    '/histogram': {
      preLoaderRoute: typeof HistogramLazyImport;
      parentRoute: typeof rootRoute;
    };
    '/thresholding': {
      preLoaderRoute: typeof ThresholdingLazyImport;
      parentRoute: typeof rootRoute;
    };
    '/hybrid': {
      preLoaderRoute: typeof HybridLazyImport;
      parentRoute: typeof rootRoute;
    };
    '/hough': {
      preLoaderRoute: typeof HoughLazyImport;
      parentRoute: typeof rootRoute;
    };
    '/contours': {
      preLoaderRoute: typeof ContoursLazyImport;
      parentRoute: typeof rootRoute;
    };
    '/corners': {
      preLoaderRoute: typeof CornersLazyImport;
      parentRoute: typeof rootRoute;
    };
    '/matching': {
      preLoaderRoute: typeof MatchingLazyImport;
      parentRoute: typeof rootRoute;
    };
    '/sift': {
      preLoaderRoute: typeof SiftLazyImport;
      parentRoute: typeof rootRoute;
    };
    '/segmentation': {
      preLoaderRoute: typeof SegmentationLazyImport;
      parentRoute: typeof rootRoute;
    };
    '/face': {
      preLoaderRoute: typeof FaceLazyImport;
      parentRoute: typeof rootRoute;
    };
  }
}

// Create and export the route tree
export const routeTree = rootRoute.addChildren([
  IndexLazyRoute,
  NoiseLazyRoute,
  FiltersLazyRoute,
  EdgeLazyRoute,
  HistogramLazyRoute,
  ThresholdingLazyRoute,
  HybridLazyRoute,
  HoughLazyRoute,
  ContoursLazyRoute,
  CornersLazyRoute,
  MatchingLazyRoute,
  SiftLazyRoute,
  SegmentationLazyRoute,
  FaceLazyRoute
]);

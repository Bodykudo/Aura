import { useTheme } from 'next-themes';
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';

export default function HistogramChart({ data }: { data: any }) {
  const { resolvedTheme } = useTheme();
  return (
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart
        width={600}
        height={300}
        data={data}
        margin={{
          top: 5,
          right: 30,
          left: 20,
          bottom: 5
        }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Area type="monotone" dataKey="red" stackId="1" stroke="red" fill="red" />
        <Area type="monotone" dataKey="green" stackId="1" stroke="green" fill="green" />
        <Area type="monotone" dataKey="blue" stackId="1" stroke="blue" fill="blue" />
        <Area
          type="monotone"
          dataKey="gray"
          stackId="1"
          stroke="black"
          fill={resolvedTheme === 'dark' ? 'white' : 'black'}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

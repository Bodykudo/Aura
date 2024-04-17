'use client';

import { Area, AreaChart, ResponsiveContainer } from 'recharts';

export default function HistogramChart({ data }: { data: any }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data}>
        <Area
          type="monotone"
          dataKey="red"
          stackId="1"
          stroke="red"
          fill="red"
        />
        <Area
          type="monotone"
          dataKey="green"
          stackId="1"
          stroke="green"
          fill="green"
        />
        <Area
          type="monotone"
          dataKey="blue"
          stackId="1"
          stroke="blue"
          fill="blue"
        />
        <Area
          type="monotone"
          dataKey="gray"
          stackId="1"
          stroke="black"
          fill="white"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('ipcRenderer', {
  send: (channel: string, data: any[]) => ipcRenderer.send(channel, data),
  on: (channel: string, func: any) => ipcRenderer.on(channel, (_, ...args) => func(...args))
});

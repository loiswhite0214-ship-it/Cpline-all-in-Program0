import { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.eagleplan.trading',
  appName: '熬鹰计划',
  webDir: 'dist',
  server: {
    androidScheme: 'https'
  },
  ios: {
    contentInset: 'automatic',
    scrollEnabled: true,
    backgroundColor: '#0f172a'
  },
  plugins: {
    StatusBar: {
      style: 'dark',
      backgroundColor: '#0f172a'
    },
    Keyboard: {
      resize: 'ionic'
    }
  }
};

export default config;

import { Box, Heading } from '@chakra-ui/react';
import ChatBot from './ChatBotModal';

function App() {
  return (
    <Box p={4}>
      <Heading as="h1" mb={4}>
        Hello from React
      </Heading>
      <ChatBot />
    </Box>
  );
}

export default App;

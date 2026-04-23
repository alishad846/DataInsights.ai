import MainLayout from '../layout/MainLayout';

const DataChatPage = () => {
  return (
    <MainLayout>
      <div style={{ padding: '2rem', textAlign: 'center' }}>
        <h2>Data Chat</h2>
        <p style={{ color: 'var(--text-muted)' }}>Chat functionality for your datasets.</p>
      </div>
    </MainLayout>
  );
};

export default DataChatPage;

(() => {
  const img = document.getElementById('stream');
  const hint = document.getElementById('hint');
  const title = document.getElementById('title-text');

  const params = new URLSearchParams(window.location.search);
  const topic = params.get('topic');
  if (topic) {
    title.textContent = `${topic} 实时画面`;
  }

  img.onload = () => {
    hint.textContent = '流已连接，正在实时刷新。';
  };

  img.onerror = () => {
    hint.textContent = '图像流连接失败，请检查后端和 Zenoh 发布端。';
  };

  img.src = '/stream.mjpg';
})();

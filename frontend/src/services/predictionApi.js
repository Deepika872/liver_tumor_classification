
import axios from "axios";

const api = axios.create({
  baseURL: "http://127.0.0.1:8000",
  timeout: 30000,
});

export async function classifyImage(file) {
  const formData = new FormData();
  formData.append("image", file);

  const response = await api.post(
    "/api/v1/predictions",
    formData
  );

  return response.data;
}

export default api;

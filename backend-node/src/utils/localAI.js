export const localBusinessAI = (question) => {
  const q = question.toLowerCase();

  if (q.includes("sales")) {
    return "To increase sales, focus on understanding customer needs, improving product value, optimizing pricing, and strengthening digital marketing channels.";
  }

  if (q.includes("marketing")) {
    return "Effective marketing strategies include targeted ads, content marketing, SEO, and customer retention programs.";
  }

  if (q.includes("customer")) {
    return "Improving customer experience, support, and trust helps increase retention and lifetime value.";
  }

  return "This question is outside the dataset scope. Please ask a data-related question or refine your business query.";
};

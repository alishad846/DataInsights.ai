// IN-MEMORY MOCK FOR NATIVE TESTING
class MockDatasetModel {
  static datasets = {};

  static async create(data) {
    const id = `mock-${Date.now()}`;
    const newDataset = { 
      _id: id, 
      ...data, 
      uploadedAt: new Date(),
      rows: data.rows || 0,
      columns: data.columns || 0,
      toObject: () => ({ _id: id, ...data, uploadedAt: new Date() }) 
    };
    this.datasets[id] = newDataset;
    return newDataset;
  }

  static async findById(id) {
    const doc = this.datasets[id] || null;
    if (doc) {
      return { 
        ...doc, 
        select: () => doc,
        toObject: () => doc 
      };
    }
    return { select: () => null }; // Quick mock for select chain
  }

  static async findByIdAndUpdate(id, update, options) {
    if (this.datasets[id]) {
      this.datasets[id] = { ...this.datasets[id], ...update };
      return this.datasets[id];
    }
    return null;
  }

  static find(filter = {}) {
    return { 
      sort: (criteria) => {
        let items = Object.values(this.datasets);
        if (filter.userId) {
          items = items.filter(d => d.userId === filter.userId);
        }
        
        const sorted = items.sort((a, b) => {
          if (criteria.uploadedAt === -1) return new Date(b.uploadedAt) - new Date(a.uploadedAt);
          return new Date(a.uploadedAt) - new Date(b.uploadedAt);
        });
        return sorted;
      }
    };
  }
}

export default MockDatasetModel;

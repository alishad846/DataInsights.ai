import Navbar from "../components/Navbar";
import Sidebar from "../components/Sidebar";
import Table from "../components/table";
import axios from "axios";
import { useEffect, useState } from "react";

export default function AdminDashboard() {
    const [data, setData] = useState([]);

    useEffect(() => {
        axios.get("http://localhost:5000/admin/dashboard", {
            headers: { authorization: sessionStorage.getItem("token") }
        }).then(res => {
            const formatted = res.data.map(d => ({
                Dataset: d.fileName,
                User: d.uploadedBy?.name,
                Date: new Date(d.uploadedAt).toLocaleString()
            }));
            setData(formatted);
        });
    }, []);

    return (
        <div>
            <Navbar title="Admin Dashboard" />
            <Sidebar />

            <div style={{ marginLeft: "220px", padding: "20px" }}>
                <Table columns={["Dataset", "User", "Date"]} data={data} />
            </div>
        </div>
    );
}

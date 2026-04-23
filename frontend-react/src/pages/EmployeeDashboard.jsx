import axios from "axios";
import { useState } from "react";

export default function EmployeeDashboard() {
    const [file, setFile] = useState(null);

    const upload = async () => {
        const form = new FormData();
        form.append("file", file);

        await axios.post("http://localhost:5000/employee/upload", form, {
            headers: { authorization: sessionStorage.getItem("token") }
        });

        alert("Uploaded");
    };

    return (
        <div style={{ padding: "20px" }}>
            <h2>Employee Dashboard</h2>

            <input type="file" onChange={e => setFile(e.target.files[0])} />
            <button onClick={upload}>Upload</button>
        </div>
    );
}

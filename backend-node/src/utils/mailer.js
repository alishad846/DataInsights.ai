import nodemailer from 'nodemailer';
import dotenv from 'dotenv';
dotenv.config();

const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
        user: process.env.EMAIL_USER,
        pass: process.env.EMAIL_PASS
    }
});

export const sendResetEmail = async (to, resetToken) => {
    try {
        const frontendUrl = process.env.FRONTEND_URL || 'http://localhost:5173';
        const resetLink = `${frontendUrl}/reset-password?token=${resetToken}`;
        
        const mailOptions = {
            from: process.env.EMAIL_USER,
            to,
            subject: 'Reset your password',
            text: `You have requested to reset your password.\n\nPlease click the link below to reset your password. This link will expire in 15 minutes.\n\n${resetLink}\n\nIf you did not request a password reset, please ignore this email.`,
            html: `
                <div style="font-family: Arial, sans-serif; padding: 20px; line-height: 1.6;">
                    <h2>Reset your password</h2>
                    <p>You have requested to reset your password.</p>
                    <p>Please click the button below to reset your password. This link will expire in 15 minutes.</p>
                    <a href="${resetLink}" style="display: inline-block; padding: 10px 20px; background-color: #58a6ff; color: #fff; text-decoration: none; border-radius: 5px; font-weight: bold;">Reset Password</a>
                    <p style="margin-top: 20px; font-size: 12px; color: #666;">Or copy and paste this link into your browser:<br>${resetLink}</p>
                    <hr style="border: none; border-top: 1px solid #eaeaea; margin: 20px 0;">
                    <p style="font-size: 12px; color: #888;">If you did not request a password reset, please ignore this email.</p>
                </div>
            `
        };

        const info = await transporter.sendMail(mailOptions);
        console.log('Reset email sent: ' + info.messageId);
        return true;
    } catch (error) {
        console.error('Error sending reset email:', error);
        return false;
    }
};
